package my.aplication.manejointeligentedepragas;

import android.app.Dialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;

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

import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.Button;
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
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Aplicacao;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PlanoAmostragem;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PresencaPraga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;
import my.aplication.manejointeligentedepragas.model.PresencaPragaModel;

import com.example.manejointeligentedepragas.R;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.zplesac.connectionbuddy.ConnectionBuddy;
import com.zplesac.connectionbuddy.ConnectionBuddyConfiguration;
import com.zplesac.connectionbuddy.interfaces.ConnectivityChangeListener;
import com.zplesac.connectionbuddy.models.ConnectivityEvent;

import java.util.ArrayList;


public class ConfirmacaoPlanoAmostragem extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{
    int codPropriedade;
    int codPraga;
    int codCultura;
    String nome;
    String nomePraga;
    boolean controla;
    boolean aplicado;
    String nomePropriedade;
    int Cod_Talhao;
    String NomeTalhao;
    int Cod_Planta;

    int codPragaComparacaoAux;

    TextView tvMostraCultura;
    TextView tvMostraPraga;
    TextView tvMostraControla;
    TextView tvMostraTalhao;

    Button btnSalvar;
    private Dialog mDialog;

    private DrawerLayout drawerLayout;

    ArrayList<PresencaPragaModel> presencaPragaModels = new ArrayList();

    ArrayList<PlanoAmostragemModel> planoAmostragemModels = new ArrayList();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_confirmacao_plano_amostragem);

        ConnectionBuddyConfiguration networkInspectorConfiguration = new ConnectionBuddyConfiguration.Builder(this).build();
        ConnectionBuddy.getInstance().init(networkInspectorConfiguration);

        openDialog();

        codPropriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        codCultura = getIntent().getIntExtra("Cod_Cultura", 0);
        nome = getIntent().getStringExtra("NomeCultura");
        nomePraga = getIntent().getStringExtra("NomePraga");
        codPraga = getIntent().getIntExtra("Cod_Praga", 0);
        controla = getIntent().getBooleanExtra("Controla",false);
        aplicado = getIntent().getBooleanExtra("Aplicado", false);
        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        Cod_Talhao = getIntent().getIntExtra("Cod_Talhao", 0);
        NomeTalhao = getIntent().getStringExtra("NomeTalhao");
        Cod_Planta = getIntent().getIntExtra("Cod_Planta",0);

        tvMostraCultura = findViewById(R.id.tvConfMostraCultura);
        tvMostraPraga = findViewById(R.id.tvConfMostraPraga);
        tvMostraControla = findViewById(R.id.tvConfMostraControle);
        tvMostraTalhao = findViewById(R.id.tvConfMostraTalhao);
        btnSalvar = findViewById(R.id.btnConfSalvaMetodo);

        tvMostraCultura.setText(nome);
        tvMostraPraga.setText(nomePraga);
        tvMostraTalhao.setText(NomeTalhao);

        buscaCodPraga(Cod_Talhao);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_confPlano);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_confPlano);
        NavigationView navigationView = findViewById(R.id.nav_view_confPlano);
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



        btnSalvar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(controla){
                    if(aplicado){
                        Intent i = new Intent(ConfirmacaoPlanoAmostragem.this, Pragas.class);
                        i.putExtra("Cod_Talhao", Cod_Talhao);
                        i.putExtra("NomeTalhao", NomeTalhao);
                        i.putExtra("Cod_Propriedade", codPropriedade);
                        i.putExtra("Cod_Cultura", codCultura);
                        i.putExtra("NomeCultura", nome);
                        i.putExtra("Aplicado", aplicado);
                        i.putExtra("nomePropriedade", nomePropriedade);
                        i.putExtra("Cod_Planta", Cod_Planta);
                        startActivity(i);
                    }else{
                        Intent i = new Intent(ConfirmacaoPlanoAmostragem.this, AplicaMetodoDeControle.class);
                        i.putExtra("Cod_Talhao", Cod_Talhao);
                        i.putExtra("NomeTalhao", NomeTalhao);
                        i.putExtra("Cod_Propriedade", codPropriedade);
                        i.putExtra("Cod_Cultura", codCultura);
                        i.putExtra("NomeCultura", nome);
                        i.putExtra("Cod_Praga",codPraga);
                        i.putExtra("Aplicado", aplicado);
                        i.putExtra("nomePropriedade", nomePropriedade);
                        i.putExtra("Cod_Planta", Cod_Planta);
                        startActivity(i);
                    }
                }else{
                    Intent i = new Intent(ConfirmacaoPlanoAmostragem.this, Pragas.class);
                    i.putExtra("Cod_Talhao", Cod_Talhao);
                    i.putExtra("NomeTalhao", NomeTalhao);
                    i.putExtra("Cod_Propriedade", codPropriedade);
                    i.putExtra("Cod_Cultura", codCultura);
                    i.putExtra("NomeCultura", nome);
                    i.putExtra("Aplicado", aplicado);
                    i.putExtra("nomePropriedade", nomePropriedade);
                    i.putExtra("Cod_Planta", Cod_Planta);
                    startActivity(i);
                }
            }
        });


    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            Intent i = new Intent(ConfirmacaoPlanoAmostragem.this, AcoesCultura.class);
            i.putExtra("Cod_Talhao", Cod_Talhao);
            i.putExtra("NomeTalhao", NomeTalhao);
            i.putExtra("Cod_Propriedade", codPropriedade);
            i.putExtra("Cod_Cultura", codCultura);
            i.putExtra("NomeCultura", nome);
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
                    final Controller_PlanoAmostragem cpa = new Controller_PlanoAmostragem(ConfirmacaoPlanoAmostragem.this);
                    final Controller_PresencaPraga cpp = new Controller_PresencaPraga(ConfirmacaoPlanoAmostragem.this);

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
        Controller_Usuario cu = new Controller_Usuario(ConfirmacaoPlanoAmostragem.this);
        String Autor = cu.getUser().getNome();

        String url = "https://mip.software/phpapp/salvaPlanoAmostragem.php?Cod_Talhao=" + pam.getFk_Cod_Talhao()
                +"&&Data="+pam.getDate()
                +"&&PlantasInfestadas="+pam.getPlantasInfestadas()
                +"&&PlantasAmostradas="+pam.getPlantasAmostradas()
                +"&&Cod_Praga="+pam.getFk_Cod_Praga()
                +"&&Autor="+Autor;

        RequestQueue queue = Volley.newRequestQueue(ConfirmacaoPlanoAmostragem.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }

        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(ConfirmacaoPlanoAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    public void SalvarPresencas(PresencaPragaModel ppm){
        String url = "https://mip.software/phpapp/updatePraga.php?Cod_Praga="+ppm.getFk_Cod_Praga()+
                "&&Cod_Talhao="+ppm.getFk_Cod_Talhao()+"&&Status="+ppm.getStatus();
        RequestQueue queue = Volley.newRequestQueue(ConfirmacaoPlanoAmostragem.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(ConfirmacaoPlanoAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
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


    public void buscaCodPraga(final int cod_Talhao){
        // selecionar o código da praga da ultima aplicação, para saber em qual praga foi aplicada e não mudar pra vermelho
        // quando já tiver alguma aplicação já realizada nela
        // mudando também o texto que aparece na tela
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            mDialog.dismiss();
            Controller_Aplicacao ca = new Controller_Aplicacao(ConfirmacaoPlanoAmostragem.this);
            codPragaComparacaoAux = ca.BuscaCodPraga(cod_Talhao);
            if (controla){
                if(aplicado){
                    if(codPragaComparacaoAux == codPraga){
                        tvMostraControla.setText("É necessário o controle, porém percebemos que uma aplicação foi realizada recentemente para esta praga. Aguarde o intervalo recomendado entre as aplicações para evitar fitotoxidez na cultura.");
                        btnSalvar.setText("OK");
                        alteraStatus(cod_Talhao, codPraga,1); //1 = amarelo
                    }else{
                        tvMostraControla.setText("É necessário o controle, porém percebemos que um método de controle foi aplicado recentemente. Aguarde o intervalo recomendado entre as aplicações para evitar fitotoxidez na cultura.");
                        btnSalvar.setText("OK");
                        alteraStatus(cod_Talhao, codPraga, 2); //2 = vermelho precisa de controle
                    }
                }else{
                    tvMostraControla.setText("É necessário o controle.");
                    btnSalvar.setText("Selecionar método");
                    alteraStatus(cod_Talhao, codPraga, 2); //2 = vermelho precisa de controle
                }
            }else{
                tvMostraControla.setText("Não é necessário realizar nenhum tipo de controle, a praga está controlada.");
                btnSalvar.setText("OK");
                //altera status praga
                alteraStatus(cod_Talhao, codPraga, 0); //0 = não precisa de controle
            }

        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/buscaCodPraga.php?Cod_Talhao="+ cod_Talhao ;
            final RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {
                @Override
                public void onResponse(String response) {
                    try {
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj1 = array.getJSONObject(i);
                            codPragaComparacaoAux = obj1.getInt("fk_Praga_Cod_Praga");
                        }
                    } catch (JSONException e) {
                        mDialog.dismiss();
                        Toast.makeText(ConfirmacaoPlanoAmostragem.this, e.toString(), Toast.LENGTH_LONG).show();
                    }

                    if (controla){
                        if(aplicado){
                            if(codPragaComparacaoAux == codPraga){
                                tvMostraControla.setText("É necessário o controle, porém percebemos que uma aplicação foi realizada recentemente para esta praga. Aguarde o intervalo recomendado entre as aplicações para evitar fitotoxidez na cultura.");
                                btnSalvar.setText("OK");
                                alteraStatus(cod_Talhao, codPraga,1); //1 = amarelo
                            }else{
                                tvMostraControla.setText("É necessário o controle, porém percebemos que um método de controle foi aplicado recentemente. Aguarde o intervalo recomendado entre as aplicações para evitar fitotoxidez na cultura.");
                                btnSalvar.setText("OK");
                                alteraStatus(cod_Talhao, codPraga, 2); //2 = vermelho precisa de controle
                            }
                        }else{
                            tvMostraControla.setText("É necessário o controle.");
                            btnSalvar.setText("Selecionar método");
                            alteraStatus(cod_Talhao, codPraga, 2); //2 = vermelho precisa de controle
                        }
                    }else{
                        tvMostraControla.setText("Não é necessário realizar nenhum tipo de controle, a praga está controlada.");
                        btnSalvar.setText("OK");
                        //altera status praga
                        alteraStatus(cod_Talhao, codPraga, 0); //0 = não precisa de controle
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(ConfirmacaoPlanoAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }
    }



    public void alteraStatus(int cod_Talhao, int codPraga, int status){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            mDialog.dismiss();
            Controller_PresencaPraga cpp = new Controller_PresencaPraga(ConfirmacaoPlanoAmostragem.this);
            cpp.updatePresencaStatus(cod_Talhao,codPraga,status);
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/alteraStatus.php?Cod_Praga=" + codPraga + "&&Cod_Talhao="+ cod_Talhao + "&&Status=" + status;
            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    mDialog.dismiss();
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    mDialog.dismiss();
                    Toast.makeText(ConfirmacaoPlanoAmostragem.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }
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
