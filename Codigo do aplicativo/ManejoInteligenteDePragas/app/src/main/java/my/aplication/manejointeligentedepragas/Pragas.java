package my.aplication.manejointeligentedepragas;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;


//import android.support.design.widget.FloatingActionButton;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

import android.os.Bundle;
//import android.support.v7.widget.LinearLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;


import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.text.style.StyleSpan;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PlanoAmostragem;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Praga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PresencaPraga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.RecyclerViewAdapter.PragaCardAdapter;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;
import my.aplication.manejointeligentedepragas.model.PragaModel;
import my.aplication.manejointeligentedepragas.model.PresencaPragaModel;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

import com.zplesac.connectionbuddy.ConnectionBuddy;
import com.zplesac.connectionbuddy.ConnectionBuddyConfiguration;
import com.zplesac.connectionbuddy.interfaces.ConnectivityChangeListener;
import com.zplesac.connectionbuddy.models.ConnectivityEvent;

public class Pragas extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    public FloatingActionButton fabAddPraga;
    public TextView tvAddPraga;
    private ArrayList<PragaModel> cards = new ArrayList<>();
    int codCultura;
    int Cod_Propriedade;
    String nomePropriedade;
    boolean aplicado;
    String nome;
    int Cod_Talhao;
    String NomeTalhao;
    int Cod_Planta;
    ArrayList<String> pragasAdd = new ArrayList<String>();
    private Dialog mDialog;
    ArrayList<PresencaPragaModel> localPresenca = new ArrayList<>();
    ArrayList<PresencaPragaModel> webPresenca = new ArrayList<>();

    ArrayList<PresencaPragaModel> presencaPragaModels = new ArrayList();
    ArrayList<PlanoAmostragemModel> planoAmostragemModels = new ArrayList();


    private DrawerLayout drawerLayout;

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_praga, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.icInfo:
                ExibirCaixaInfo();
                return true;
        }

        return super.onOptionsItemSelected(item);
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pragas);

        ConnectionBuddyConfiguration networkInspectorConfiguration = new ConnectionBuddyConfiguration.Builder(this).build();
        ConnectionBuddy.getInstance().init(networkInspectorConfiguration);

        openDialog();

        Cod_Talhao = getIntent().getIntExtra("Cod_Talhao", 0);
        NomeTalhao = getIntent().getStringExtra("NomeTalhao");
        aplicado = getIntent().getBooleanExtra("Aplicado", false);
        codCultura = getIntent().getIntExtra("Cod_Cultura", 0);
        nome = getIntent().getStringExtra("NomeCultura");
        Cod_Propriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        Cod_Planta = getIntent().getIntExtra("Cod_Planta",0);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_praga);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_pragas);
        NavigationView navigationView = findViewById(R.id.nav_view_praga);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);

        setTitle("MIP² | "+nome+": "+NomeTalhao);

        resgatarDados();

        fabAddPraga = findViewById(R.id.fabAddPraga);
        fabAddPraga.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(Pragas.this, AdicionarPraga.class);
                i.putExtra("Cod_Talhao", Cod_Talhao);
                i.putExtra("NomeTalhao", NomeTalhao);
                i.putExtra("Cod_Cultura", codCultura);
                i.putExtra("NomeCultura", nome);
                i.putExtra("Cod_Propriedade", Cod_Propriedade);
                i.putExtra("pragasAdd", pragasAdd);
                i.putExtra("Aplicado", aplicado);
                i.putExtra("Cod_Planta", Cod_Planta);
                startActivity(i);
            }
        });

        tvAddPraga = findViewById(R.id.tvAddPraga);
        tvAddPraga.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(Pragas.this, AdicionarPraga.class);
                i.putExtra("Cod_Talhao", Cod_Talhao);
                i.putExtra("NomeTalhao", NomeTalhao);
                i.putExtra("Cod_Cultura", codCultura);
                i.putExtra("NomeCultura", nome);
                i.putExtra("Cod_Propriedade", Cod_Propriedade);
                i.putExtra("pragasAdd", pragasAdd);
                i.putExtra("Aplicado", aplicado);
                i.putExtra("nomePropriedade", nomePropriedade);
                i.putExtra("Cod_Planta", Cod_Planta);
                startActivity(i);
            }
        });


    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            Intent i = new Intent(Pragas.this, AcoesCultura.class);
            i.putExtra("Cod_Talhao", Cod_Talhao);
            i.putExtra("NomeTalhao", NomeTalhao);
            i.putExtra("Cod_Cultura", codCultura);
            i.putExtra("NomeCultura", nome);
            i.putExtra("Cod_Propriedade", Cod_Propriedade);
            i.putExtra("Aplicado", aplicado);
            i.putExtra("nomePropriedade", nomePropriedade);
            i.putExtra("Cod_Planta", Cod_Planta);
            startActivity(i);
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        ConnectionBuddy.getInstance().registerForConnectivityEvents(this, new ConnectivityChangeListener() {
            @Override
            public void onConnectionChange(ConnectivityEvent event) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    //Toast.makeText(AcoesCultura.this,"Você está offline!", Toast.LENGTH_LONG).show();
                }else{
                    final Controller_PlanoAmostragem cpa = new Controller_PlanoAmostragem(Pragas.this);
                    final Controller_PresencaPraga cpp = new Controller_PresencaPraga(Pragas.this);

                    //Toast.makeText(AcoesCultura.this,"Você está online!", Toast.LENGTH_LONG).show();

                    planoAmostragemModels = cpa.getPlanoOffline();
                    presencaPragaModels = cpp.getPresencaPragaOffline();

                    for(int i=0; i<planoAmostragemModels.size(); i++){
                        SalvarPlanos(planoAmostragemModels.get(i));
                    }
                    cpa.removerPlano();

                    for(int i=0; i<presencaPragaModels.size(); i++){
                        SalvarPresencas(presencaPragaModels.get(i));
                    }
                    cpp.updatePresencaSyncStatus();


                }
            }
        });
    }

    @Override
    protected void onStop() {
        super.onStop();
        ConnectionBuddy.getInstance().unregisterFromConnectivityEvents(this);
    }


    public void SalvarPlanos(PlanoAmostragemModel pam){
        Controller_Usuario cu = new Controller_Usuario(Pragas.this);
        String Autor = cu.getUser().getNome();

        String url = "https://mip.software/phpapp/salvaPlanoAmostragem.php?Cod_Talhao=" + pam.getFk_Cod_Talhao()
                +"&&Data="+pam.getDate()
                +"&&PlantasInfestadas="+pam.getPlantasInfestadas()
                +"&&PlantasAmostradas="+pam.getPlantasAmostradas()
                +"&&Cod_Praga="+pam.getFk_Cod_Praga()
                +"&&Autor="+Autor;

        RequestQueue queue = Volley.newRequestQueue(Pragas.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }

        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(Pragas.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    public void SalvarPresencas(PresencaPragaModel ppm){
        String url = "https://mip.software/phpapp/updatePraga.php?Cod_Praga="+ppm.getFk_Cod_Praga()+
                "&&Cod_Talhao="+ppm.getFk_Cod_Talhao()+"&&Status="+ppm.getStatus();
        RequestQueue queue = Volley.newRequestQueue(Pragas.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(Pragas.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Intent i= new Intent(this, Perfil.class);
                startActivity(i);
                break;
            case R.id.drawerProp:
                Intent prop= new Intent(this, Propriedades.class);
                startActivity(prop);
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;

            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }


    private void iniciarRecyclerView() {
        RecyclerView rv = findViewById(R.id.RVPraga);
        PragaCardAdapter adapter = new PragaCardAdapter(this, cards, Cod_Talhao, NomeTalhao, codCultura, nome, Cod_Propriedade, aplicado, nomePropriedade, Cod_Planta);

        rv.setAdapter(adapter);
        rv.setLayoutManager(new LinearLayoutManager(this));

    }

    private void resgatarDados() {
        //Log.d(TAG, "resgatarDados: resgatou");
        final Controller_PresencaPraga cpp = new Controller_PresencaPraga(Pragas.this);
        Controller_Praga cp = new Controller_Praga(Pragas.this);

        //Utils u = new Utils();
        //if (!u.isConected(getBaseContext())) {

            localPresenca = cpp.getPresencaPraga(Cod_Talhao);
            for(int i=0; i<localPresenca.size();i++){
               PragaModel pm = new PragaModel();
               pm.setCod_Praga(localPresenca.get(i).getFk_Cod_Praga());
               pm.setNome(cp.getNome(localPresenca.get(i).getFk_Cod_Praga()));
               pm.setStatus(localPresenca.get(i).getStatus());
               cards.add(pm);
               pragasAdd.add(pm.getNome());
            }
            iniciarRecyclerView();
            mDialog.dismiss();
        /*} else { // se tem acesso à internet

            String url = "https://mip.software/phpapp/resgatarPragas.php?Cod_Talhao=" + Cod_Talhao;

            RequestQueue queue = Volley.newRequestQueue(Pragas.this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        cpp.removerPresencaPraga();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i < array.length(); i++) {
                            JSONObject obj = array.getJSONObject(i);
                            PragaModel u = new PragaModel();
                            u.setCod_Praga(obj.getInt("Cod_Praga"));
                            u.setNome(obj.getString("Nome"));
                            u.setStatus(obj.getInt("Status"));
                            int codPresenca = obj.getInt("Cod_PresencaPraga");
                            cards.add(u);
                            pragasAdd.add(obj.getString("Nome"));
                            cpp.addPresencaSemCod(u.getStatus(), Cod_Talhao, u.getCod_Praga(), 0);
                        }
                        iniciarRecyclerView();
                        mDialog.dismiss();
                    } catch (JSONException e) {
                        Toast.makeText(Pragas.this, e.toString(), Toast.LENGTH_LONG).show();
                        mDialog.dismiss();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(Pragas.this, error.toString(), Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }
            }));

        }*/

    }

    public  void ExibirCaixaInfo(){
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(Pragas.this);
        SpannableString ss =
                new SpannableString("As cores nessa tela indicam diferentes situações:\n\n" +
                "Verde: indica que a praga encontra-se controlada no momento (abaixo do nível de controle)." +
                "\n\nAmarelo: indica que é necessário realizar uma amostragem sobre a praga." +
                "\n\nVermelho: indica que, após uma contagem, foi constatada a necessidade de aplicação de algum método de controle.");
        ForegroundColorSpan foregroundVerde = new ForegroundColorSpan(Color.parseColor("#659251"));
        ForegroundColorSpan foregroundAmarelo = new ForegroundColorSpan(Color.parseColor("#ECC911"));
        ForegroundColorSpan foregroundVermelho = new ForegroundColorSpan(Color.parseColor("#FD991111"));
        ss.setSpan(foregroundVerde, 51,58, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        ss.setSpan(foregroundAmarelo, 143,151, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        ss.setSpan(foregroundVermelho, 215,225, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);

        StyleSpan negrito = new StyleSpan(Typeface.BOLD);
        StyleSpan negrito2 = new StyleSpan(Typeface.BOLD);
        StyleSpan negrito3 = new StyleSpan(Typeface.BOLD);
        ss.setSpan(negrito3, 51,58, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        ss.setSpan(negrito2, 143,151, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        ss.setSpan(negrito, 215,225, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);

        dlgBox.setTitle("Informações");
        dlgBox.setMessage(ss);

        /*dlgBox.setMessage("As cores nessa tela indicam diferentes situações:\n\n" +
                          "Verde: indica que a praga encontra-se controlada no momento(abaixo do nível de controle)." +
                          "\n\nAmarelo: indica que é necessário realizar uma amostragem sobre a praga." +
                          "\n\nVermelho: indica que, após uma contagem, foi constatada a necessidade de aplicação de algum método de controle.");
        */dlgBox.setPositiveButton("Entendi", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {

            }
        });

        dlgBox.show();
    }

    public void openDialog(){
        mDialog = new Dialog(this);
        //vamos remover o titulo da Dialog
        mDialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        //vamos carregar o xml personalizado
        mDialog.setContentView(R.layout.dialog);
        //DEixamos transparente
        mDialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        // não permitimos fechar esta dialog
        mDialog.setCancelable(false);
        //temos a instancia do ProgressBar!
        final ProgressBar progressBar = ProgressBar.class.cast(mDialog.findViewById(R.id.progressBar));

        mDialog.show();

        // mDialog.dismiss(); -> para fechar a dialog

    }
}
