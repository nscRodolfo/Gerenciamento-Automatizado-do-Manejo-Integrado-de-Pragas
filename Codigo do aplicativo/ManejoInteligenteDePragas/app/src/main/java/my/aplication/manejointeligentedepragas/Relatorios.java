package my.aplication.manejointeligentedepragas;

import android.content.Intent;
import android.content.SharedPreferences;

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
import android.view.MenuItem;
import android.view.View;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PlanoAmostragem;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PresencaPraga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;
import my.aplication.manejointeligentedepragas.model.PresencaPragaModel;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.manejointeligentedepragas.R;

import com.zplesac.connectionbuddy.ConnectionBuddy;
import com.zplesac.connectionbuddy.ConnectionBuddyConfiguration;
import com.zplesac.connectionbuddy.interfaces.ConnectivityChangeListener;
import com.zplesac.connectionbuddy.models.ConnectivityEvent;

import java.util.ArrayList;

public class Relatorios extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener
        {

    int Cod_Propriedade;
    int codCultura;
    String nome;
    boolean aplicado;
    String nomePropriedade;
    int Cod_Talhao;
    String NomeTalhao;
    int Cod_Planta;

    RelativeLayout rlPragasContagem;
    RelativeLayout rlPlanosAmostragem;
    RelativeLayout rlCaldasAplicadas;
    RelativeLayout rlRelatoriosAplicacoesContagens;

    private DrawerLayout drawerLayout;

            ArrayList<PresencaPragaModel> presencaPragaModels = new ArrayList();

            ArrayList<PlanoAmostragemModel> planoAmostragemModels = new ArrayList();


            @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_relatorios);

                ConnectionBuddyConfiguration networkInspectorConfiguration = new ConnectionBuddyConfiguration.Builder(this).build();
                ConnectionBuddy.getInstance().init(networkInspectorConfiguration);

        Cod_Propriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        codCultura = getIntent().getIntExtra("Cod_Cultura", 0);
        nome = getIntent().getStringExtra("NomeCultura");
        aplicado = getIntent().getBooleanExtra("Aplicado", false);
        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        Cod_Talhao = getIntent().getIntExtra("Cod_Talhao", 0);
        NomeTalhao = getIntent().getStringExtra("NomeTalhao");
        Cod_Planta = getIntent().getIntExtra("Cod_Planta",0);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_relatorio);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_relatorio);
        NavigationView navigationView = findViewById(R.id.nav_view_relatorios);
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


        setTitle("MIP² | Relatórios "+nome+": "+NomeTalhao);
        rlPragasContagem = findViewById(R.id.rlRelatoriosPragasContagens);
        rlPlanosAmostragem = findViewById(R.id.rlRelatoriosPlanosDeAmostragem);
        rlCaldasAplicadas = findViewById(R.id.rlCaldasAplicadas);
        rlRelatoriosAplicacoesContagens = findViewById(R.id.rlRelatoriosAplicacoesContagens);

        rlPragasContagem.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(Relatorios.this, SelecionaPragaRelatorioPragaPlano.class);
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
        });

        rlPlanosAmostragem.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(Relatorios.this, SelecionaPragaPlanosRealizados.class);
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
        });

        rlCaldasAplicadas.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //intent aplicações realizadas na cultura
                Intent i = new Intent(Relatorios.this, SelecionaPragaRelatorioAplicacoesRealizadas.class);
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
        });

        rlRelatoriosAplicacoesContagens.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // intent grafico aplicações
                Intent i = new Intent(Relatorios.this, SelecionaPragaRelatorioAplicacoesPlanos.class);
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
        });
    }

            @Override
            public void onBackPressed() {
                if(drawerLayout.isDrawerOpen(GravityCompat.START)){
                    drawerLayout.closeDrawer(GravityCompat.START);
                }else {
                    super.onBackPressed();
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
                            final Controller_PlanoAmostragem cpa = new Controller_PlanoAmostragem(Relatorios.this);
                            final Controller_PresencaPraga cpp = new Controller_PresencaPraga(Relatorios.this);

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
                Controller_Usuario cu = new Controller_Usuario(Relatorios.this);
                String Autor = cu.getUser().getNome();

                String url = "https://mip.software/phpapp/salvaPlanoAmostragem.php?Cod_Talhao=" + pam.getFk_Cod_Talhao()
                        +"&&Data="+pam.getDate()
                        +"&&PlantasInfestadas="+pam.getPlantasInfestadas()
                        +"&&PlantasAmostradas="+pam.getPlantasAmostradas()
                        +"&&Cod_Praga="+pam.getFk_Cod_Praga()
                        +"&&Autor="+Autor;

                RequestQueue queue = Volley.newRequestQueue(Relatorios.this);
                queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                    @Override
                    public void onResponse(String response) {


                    }

                }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Toast.makeText(Relatorios.this,error.toString(), Toast.LENGTH_LONG).show();
                    }
                }));
            }

            public void SalvarPresencas(PresencaPragaModel ppm){
                String url = "https://mip.software/phpapp/updatePraga.php?Cod_Praga="+ppm.getFk_Cod_Praga()+
                        "&&Cod_Talhao="+ppm.getFk_Cod_Talhao()+"&&Status="+ppm.getStatus();
                RequestQueue queue = Volley.newRequestQueue(Relatorios.this);
                queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                    @Override
                    public void onResponse(String response) {


                    }
                }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Toast.makeText(Relatorios.this,error.toString(), Toast.LENGTH_LONG).show();
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
}
